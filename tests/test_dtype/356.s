
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n4>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s0, s6, s15                                      // 000000001618: 80000F06
	s_addc_u32 s1, s7, s3                                      // 00000000161C: 82010307
	global_load_i8 v1, v0, s[0:1]                              // 000000001620: DC460000 01000000
	s_waitcnt vmcnt(0)                                         // 000000001628: BF8903F7
	v_cmp_gt_i32_e32 vcc_lo, 1, v1                             // 00000000162C: 7C880281
	v_cvt_f32_i32_e32 v2, v1                                   // 000000001630: 7E040B01
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, vcc_lo              // 000000001634: D5010001 01A9FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001640: BF870091
	v_mul_f32_e32 v1, v1, v2                                   // 000000001644: 10020501
	v_sqrt_f32_e32 v2, v1                                      // 000000001648: 7E046701
	s_waitcnt_depctr 0xfff                                     // 00000000164C: BF880FFF
	v_add_nc_u32_e32 v4, 1, v2                                 // 000000001650: 4A080481
	v_add_nc_u32_e32 v3, -1, v2                                // 000000001654: 4A0604C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001658: BF870112
	v_fma_f32 v6, -v4, v2, v1                                  // 00000000165C: D6130006 24060504
	v_fma_f32 v5, -v3, v2, v1                                  // 000000001664: D6130005 24060503
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000166C: BF870091
	v_cmp_ge_f32_e64 s0, 0, v5                                 // 000000001670: D4160000 00020A80
	v_cndmask_b32_e64 v2, v2, v3, s0                           // 000000001678: D5010002 00020702
	v_cndmask_b32_e64 v3, 1.0, 0x37800000, vcc_lo              // 000000001680: D5010003 01A9FEF2 37800000
	v_cmp_lt_f32_e32 vcc_lo, 0, v6                             // 00000000168C: 7C220C80
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001690: 84808202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001694: BF870149
	s_add_u32 s0, s4, s0                                       // 000000001698: 80000004
	s_addc_u32 s1, s5, s1                                      // 00000000169C: 82010105
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo                       // 0000000016A0: 02040902
	v_cmp_class_f32_e64 vcc_lo, v1, 0x260                      // 0000000016A4: D47E006A 0001FF01 00000260
	v_mul_f32_e32 v2, v3, v2                                   // 0000000016B0: 10040503
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016B4: BF870001
	v_cndmask_b32_e32 v1, v2, v1, vcc_lo                       // 0000000016B8: 02020302
	global_store_b32 v0, v1, s[0:1]                            // 0000000016BC: DC6A0000 00000100
	s_nop 0                                                    // 0000000016C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016C8: BFB60003
	s_endpgm                                                   // 0000000016CC: BFB00000
