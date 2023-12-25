
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_60>:
	s_load_b128 s[16:19], s[0:1], null                         // 000000001600: F4080400 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_mov_b64 s[20:21], 0                                      // 00000000160C: BE940180
	s_waitcnt lgkmcnt(0)                                       // 000000001610: BF89FC07
	s_add_u32 s22, s18, s20                                    // 000000001614: 80161412
	s_addc_u32 s23, s19, s21                                   // 000000001618: 82171513
	s_add_u32 s20, s20, 0x50                                   // 00000000161C: 8014FF14 00000050
	s_load_b512 s[0:15], s[22:23], null                        // 000000001624: F410000B F8000000
	s_addc_u32 s21, s21, 0                                     // 00000000162C: 82158015
	s_cmpk_eq_i32 s20, 0xf0                                    // 000000001630: B19400F0
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_max_f32_e64 v1, s0, s0                                   // 000000001638: D5100001 00000000
	v_max_f32_e64 v2, s1, s1                                   // 000000001640: D5100002 00000201
	v_max_f32_e64 v3, s2, s2                                   // 000000001648: D5100003 00000402
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870092
	v_dual_max_f32 v1, 0, v1 :: v_dual_max_f32 v2, 0, v2       // 000000001654: CA940280 01020480
	v_dual_max_f32 v3, 0, v3 :: v_dual_add_f32 v0, v0, v1      // 00000000165C: CA880680 03000300
	v_max_f32_e64 v1, s3, s3                                   // 000000001664: D5100001 00000603
	s_load_b128 s[0:3], s[22:23], 0x40                         // 00000000166C: F408000B F8000040
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001674: BF870121
	v_dual_add_f32 v0, v0, v2 :: v_dual_max_f32 v1, 0, v1      // 000000001678: C9140500 00000280
	v_max_f32_e64 v2, s4, s4                                   // 000000001680: D5100002 00000804
	v_add_f32_e32 v0, v0, v3                                   // 000000001688: 06000700
	v_max_f32_e64 v3, s5, s5                                   // 00000000168C: D5100003 00000A05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001694: BF870193
	v_max_f32_e32 v2, 0, v2                                    // 000000001698: 20040480
	v_add_f32_e32 v0, v0, v1                                   // 00000000169C: 06000300
	v_max_f32_e64 v1, s6, s6                                   // 0000000016A0: D5100001 00000C06
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000016A8: BF870122
	v_dual_max_f32 v3, 0, v3 :: v_dual_add_f32 v0, v0, v2      // 0000000016AC: CA880680 03000500
	v_max_f32_e64 v2, s7, s7                                   // 0000000016B4: D5100002 00000E07
	v_dual_max_f32 v1, 0, v1 :: v_dual_add_f32 v0, v0, v3      // 0000000016BC: CA880280 01000700
	v_max_f32_e64 v3, s8, s8                                   // 0000000016C4: D5100003 00001008
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000016CC: BF870193
	v_max_f32_e32 v2, 0, v2                                    // 0000000016D0: 20040480
	v_add_f32_e32 v0, v0, v1                                   // 0000000016D4: 06000300
	v_max_f32_e64 v1, s9, s9                                   // 0000000016D8: D5100001 00001209
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000016E0: BF870122
	v_dual_max_f32 v3, 0, v3 :: v_dual_add_f32 v0, v0, v2      // 0000000016E4: CA880680 03000500
	v_max_f32_e64 v2, s10, s10                                 // 0000000016EC: D5100002 0000140A
	v_dual_max_f32 v1, 0, v1 :: v_dual_add_f32 v0, v0, v3      // 0000000016F4: CA880280 01000700
	v_max_f32_e64 v3, s11, s11                                 // 0000000016FC: D5100003 0000160B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001704: BF870193
	v_max_f32_e32 v2, 0, v2                                    // 000000001708: 20040480
	v_add_f32_e32 v0, v0, v1                                   // 00000000170C: 06000300
	v_max_f32_e64 v1, s12, s12                                 // 000000001710: D5100001 0000180C
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001718: BF870122
	v_dual_max_f32 v3, 0, v3 :: v_dual_add_f32 v0, v0, v2      // 00000000171C: CA880680 03000500
	v_max_f32_e64 v2, s13, s13                                 // 000000001724: D5100002 00001A0D
	v_dual_max_f32 v1, 0, v1 :: v_dual_add_f32 v0, v0, v3      // 00000000172C: CA880280 01000700
	v_max_f32_e64 v3, s14, s14                                 // 000000001734: D5100003 00001C0E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000173C: BF870193
	v_max_f32_e32 v2, 0, v2                                    // 000000001740: 20040480
	v_add_f32_e32 v0, v0, v1                                   // 000000001744: 06000300
	v_max_f32_e64 v1, s15, s15                                 // 000000001748: D5100001 00001E0F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001750: BF870132
	v_dual_max_f32 v3, 0, v3 :: v_dual_add_f32 v0, v0, v2      // 000000001754: CA880680 03000500
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	v_max_f32_e64 v2, s0, s0                                   // 000000001760: D5100002 00000000
	v_dual_max_f32 v1, 0, v1 :: v_dual_add_f32 v0, v0, v3      // 000000001768: CA880280 01000700
	v_max_f32_e64 v3, s1, s1                                   // 000000001770: D5100003 00000201
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001778: BF870113
	v_max_f32_e32 v2, 0, v2                                    // 00000000177C: 20040480
	v_dual_add_f32 v0, v0, v1 :: v_dual_max_f32 v3, 0, v3      // 000000001780: C9140300 00020680
	v_max_f32_e64 v1, s2, s2                                   // 000000001788: D5100001 00000402
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001790: BF870122
	v_add_f32_e32 v0, v0, v2                                   // 000000001794: 06000500
	v_max_f32_e64 v2, s3, s3                                   // 000000001798: D5100002 00000603
	v_dual_max_f32 v1, 0, v1 :: v_dual_add_f32 v0, v0, v3      // 0000000017A0: CA880280 01000700
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017A8: BF870112
	v_max_f32_e32 v2, 0, v2                                    // 0000000017AC: 20040480
	v_add_f32_e32 v0, v0, v1                                   // 0000000017B0: 06000300
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017B4: BF870001
	v_add_f32_e32 v0, v0, v2                                   // 0000000017B8: 06000500
	s_cbranch_scc0 65428                                       // 0000000017BC: BFA1FF94 <r_60+0x10>
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C0: BF870091
	v_dual_max_f32 v0, v0, v0 :: v_dual_mov_b32 v1, 0          // 0000000017C4: CA900100 00000080
	v_max_f32_e32 v0, 0, v0                                    // 0000000017CC: 20000080
	global_store_b32 v1, v0, s[16:17]                          // 0000000017D0: DC6A0000 00100001
	s_nop 0                                                    // 0000000017D8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017DC: BFB60003
	s_endpgm                                                   // 0000000017E0: BFB00000
