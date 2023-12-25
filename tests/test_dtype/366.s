
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n14>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_load_b32 s2, s[2:3], null                                // 000000001624: F4000081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_cmp_lt_i32 s2, 1                                         // 000000001630: BF048102
	v_cvt_f32_i32_e32 v0, s2                                   // 000000001634: 7E000A02
	s_cselect_b32 s2, -1, 0                                    // 000000001638: 980280C1
	s_add_u32 s0, s0, s4                                       // 00000000163C: 80000400
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, s2                  // 000000001640: D5010001 0009FEF2 4F800000
	s_addc_u32 s1, s1, s5                                      // 00000000164C: 82010501
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000001654: 10000101
	v_sqrt_f32_e32 v1, v0                                      // 000000001658: 7E026700
	s_waitcnt_depctr 0xfff                                     // 00000000165C: BF880FFF
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000001660: 4A060281
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001664: 4A0402C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001668: BF870112
	v_fma_f32 v5, -v3, v1, v0                                  // 00000000166C: D6130005 24020303
	v_fma_f32 v4, -v2, v1, v0                                  // 000000001674: D6130004 24020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 00000000167C: BF870221
	v_cmp_ge_f32_e32 vcc_lo, 0, v4                             // 000000001680: 7C2C0880
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo                       // 000000001684: 02020501
	v_cmp_lt_f32_e32 vcc_lo, 0, v5                             // 000000001688: 7C220A80
	v_cndmask_b32_e64 v2, 1.0, 0x37800000, s2                  // 00000000168C: D5010002 0009FEF2 37800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001698: BF870123
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo                       // 00000000169C: 02020701
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 0000000016A0: D47E006A 0001FF00 00000260
	v_mul_f32_e32 v1, v2, v1                                   // 0000000016AC: 10020302
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016B0: BF870001
	v_dual_cndmask_b32 v0, v1, v0 :: v_dual_mov_b32 v1, 0      // 0000000016B4: CA500101 00000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000016BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000016C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016C8: BFB60003
	s_endpgm                                                   // 0000000016CC: BFB00000
