
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n64>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[6:7], s[4:5], 3                               // 000000001614: 84868304
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b64 s[2:3], s[2:3], null                            // 000000001624: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	s_xor_b32 s6, s2, s3                                       // 000000001630: 8D060302
	s_cls_i32 s7, s3                                           // 000000001634: BE870C03
	s_ashr_i32 s6, s6, 31                                      // 000000001638: 86069F06
	s_add_i32 s7, s7, -1                                       // 00000000163C: 8107C107
	s_add_i32 s6, s6, 32                                       // 000000001640: 8106A006
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001644: BF870499
	s_min_u32 s8, s7, s6                                       // 000000001648: 89880607
	s_lshl_b64 s[6:7], s[2:3], s8                              // 00000000164C: 84860802
	v_cmp_lt_i64_e64 s2, s[2:3], 1                             // 000000001650: D4510002 00010202
	s_min_u32 s6, s6, 1                                        // 000000001658: 89868106
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000165C: BF870499
	s_or_b32 s6, s7, s6                                        // 000000001660: 8C060607
	v_cvt_f32_i32_e32 v0, s6                                   // 000000001664: 7E000A06
	s_sub_i32 s6, 32, s8                                       // 000000001668: 818608A0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000166C: BF870112
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, s2                  // 000000001670: D5010001 0009FEF2 4F800000
	v_ldexp_f32 v0, v0, s6                                     // 00000000167C: D71C0000 00000D00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001684: BF870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000001688: 10000101
	v_sqrt_f32_e32 v1, v0                                      // 00000000168C: 7E026700
	s_waitcnt_depctr 0xfff                                     // 000000001690: BF880FFF
	v_add_nc_u32_e32 v3, 1, v1                                 // 000000001694: 4A060281
	v_add_nc_u32_e32 v2, -1, v1                                // 000000001698: 4A0402C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000169C: BF870112
	v_fma_f32 v5, -v3, v1, v0                                  // 0000000016A0: D6130005 24020303
	v_fma_f32 v4, -v2, v1, v0                                  // 0000000016A8: D6130004 24020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000016B0: BF870221
	v_cmp_ge_f32_e32 vcc_lo, 0, v4                             // 0000000016B4: 7C2C0880
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo                       // 0000000016B8: 02020501
	v_cmp_lt_f32_e32 vcc_lo, 0, v5                             // 0000000016BC: 7C220A80
	v_cndmask_b32_e64 v2, 1.0, 0x37800000, s2                  // 0000000016C0: D5010002 0009FEF2 37800000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 0000000016CC: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000016D0: BF870149
	s_add_u32 s0, s0, s2                                       // 0000000016D4: 80000200
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo                       // 0000000016D8: 02020701
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 0000000016DC: D47E006A 0001FF00 00000260
	s_addc_u32 s1, s1, s3                                      // 0000000016E8: 82010301
	v_mul_f32_e32 v1, v2, v1                                   // 0000000016EC: 10020302
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016F0: BF870001
	v_dual_cndmask_b32 v0, v1, v0 :: v_dual_mov_b32 v1, 0      // 0000000016F4: CA500101 00000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000016FC: DC6A0000 00000001
	s_nop 0                                                    // 000000001704: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001708: BFB60003
	s_endpgm                                                   // 00000000170C: BFB00000
