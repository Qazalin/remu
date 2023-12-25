
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_10_10n2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s6, s15                                          // 000000001608: BE86000F
	s_ashr_i32 s7, s15, 31                                     // 00000000160C: 86079F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[4:5], s[6:7], 2                               // 000000001614: 84848206
	s_add_i32 s7, s15, -1                                      // 000000001618: 8107C10F
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s2, s2, s4                                       // 000000001620: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001624: 82030503
	s_add_u32 s2, s2, 0xffffffdc                               // 000000001628: 8002FF02 FFFFFFDC
	s_addc_u32 s3, s3, -1                                      // 000000001630: 8203C103
	s_cmp_lt_i32 s7, 8                                         // 000000001634: BF048807
	s_cbranch_scc0 73                                          // 000000001638: BFA10049 <r_10_10n2+0x160>
	s_mov_b32 s7, 0                                            // 00000000163C: BE870080
	s_cmp_lt_i32 s6, 8                                         // 000000001640: BF048806
	s_mov_b32 s8, 0                                            // 000000001644: BE880080
	s_cbranch_scc0 78                                          // 000000001648: BFA1004E <r_10_10n2+0x184>
	s_add_i32 s9, s6, 1                                        // 00000000164C: 81098106
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001650: BF870009
	s_cmp_lt_i32 s9, 8                                         // 000000001654: BF048809
	s_cbranch_scc1 2                                           // 000000001658: BFA20002 <r_10_10n2+0x64>
	s_load_b32 s7, s[2:3], 0x8                                 // 00000000165C: F40001C1 F8000008
	s_add_i32 s10, s6, 2                                       // 000000001664: 810A8206
	s_mov_b32 s9, 0                                            // 000000001668: BE890080
	s_cmp_lt_i32 s10, 8                                        // 00000000166C: BF04880A
	s_mov_b32 s10, 0                                           // 000000001670: BE8A0080
	s_cbranch_scc1 2                                           // 000000001674: BFA20002 <r_10_10n2+0x80>
	s_load_b32 s10, s[2:3], 0xc                                // 000000001678: F4000281 F800000C
	s_add_i32 s11, s6, 3                                       // 000000001680: 810B8306
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001684: BF870009
	s_cmp_lt_i32 s11, 8                                        // 000000001688: BF04880B
	s_cbranch_scc1 2                                           // 00000000168C: BFA20002 <r_10_10n2+0x98>
	s_load_b32 s9, s[2:3], 0x10                                // 000000001690: F4000241 F8000010
	s_add_i32 s12, s6, 4                                       // 000000001698: 810C8406
	s_mov_b32 s11, 0                                           // 00000000169C: BE8B0080
	s_cmp_lt_i32 s12, 8                                        // 0000000016A0: BF04880C
	s_mov_b32 s12, 0                                           // 0000000016A4: BE8C0080
	s_cbranch_scc1 2                                           // 0000000016A8: BFA20002 <r_10_10n2+0xb4>
	s_load_b32 s12, s[2:3], 0x14                               // 0000000016AC: F4000301 F8000014
	s_add_i32 s13, s6, 5                                       // 0000000016B4: 810D8506
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000016B8: BF870009
	s_cmp_lt_i32 s13, 8                                        // 0000000016BC: BF04880D
	s_cbranch_scc1 2                                           // 0000000016C0: BFA20002 <r_10_10n2+0xcc>
	s_load_b32 s11, s[2:3], 0x18                               // 0000000016C4: F40002C1 F8000018
	s_add_i32 s14, s6, 6                                       // 0000000016CC: 810E8606
	s_mov_b32 s13, 0                                           // 0000000016D0: BE8D0080
	s_cmp_lt_i32 s14, 8                                        // 0000000016D4: BF04880E
	s_mov_b32 s14, 0                                           // 0000000016D8: BE8E0080
	s_cbranch_scc0 48                                          // 0000000016DC: BFA10030 <r_10_10n2+0x1a0>
	s_add_i32 s15, s6, 7                                       // 0000000016E0: 810F8706
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000016E4: BF870009
	s_cmp_lt_i32 s15, 8                                        // 0000000016E8: BF04880F
	s_cbranch_scc0 50                                          // 0000000016EC: BFA10032 <r_10_10n2+0x1b8>
	s_cmp_gt_u32 s6, 0x7ffffff7                                // 0000000016F0: BF08FF06 7FFFFFF7
	s_mov_b32 s6, 0                                            // 0000000016F8: BE860080
	s_cbranch_scc1 2                                           // 0000000016FC: BFA20002 <r_10_10n2+0x108>
	s_load_b32 s6, s[2:3], 0x24                                // 000000001700: F4000181 F8000024
	s_waitcnt lgkmcnt(0)                                       // 000000001708: BF89FC07
	v_dual_add_f32 v0, s8, v0 :: v_dual_mov_b32 v1, 0          // 00000000170C: C9100008 00000080
	s_add_u32 s0, s0, s4                                       // 000000001714: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001718: 82010501
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000171C: BF870091
	v_add_f32_e32 v0, s7, v0                                   // 000000001720: 06000007
	v_add_f32_e32 v0, s10, v0                                  // 000000001724: 0600000A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001728: BF870091
	v_add_f32_e32 v0, s9, v0                                   // 00000000172C: 06000009
	v_add_f32_e32 v0, s12, v0                                  // 000000001730: 0600000C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001734: BF870091
	v_add_f32_e32 v0, s11, v0                                  // 000000001738: 0600000B
	v_add_f32_e32 v0, s14, v0                                  // 00000000173C: 0600000E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001740: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 000000001744: 0600000D
	v_add_f32_e32 v0, s6, v0                                   // 000000001748: 06000006
	global_store_b32 v1, v0, s[0:1]                            // 00000000174C: DC6A0000 00000001
	s_nop 0                                                    // 000000001754: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001758: BFB60003
	s_endpgm                                                   // 00000000175C: BFB00000
	s_load_b32 s7, s[2:3], null                                // 000000001760: F40001C1 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001768: BF89FC07
	v_add_f32_e64 v0, s7, 0                                    // 00000000176C: D5030000 00010007
	s_mov_b32 s7, 0                                            // 000000001774: BE870080
	s_cmp_lt_i32 s6, 8                                         // 000000001778: BF048806
	s_mov_b32 s8, 0                                            // 00000000177C: BE880080
	s_cbranch_scc1 65458                                       // 000000001780: BFA2FFB2 <r_10_10n2+0x4c>
	s_load_b32 s8, s[2:3], 0x4                                 // 000000001784: F4000201 F8000004
	s_add_i32 s9, s6, 1                                        // 00000000178C: 81098106
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001790: BF870009
	s_cmp_lt_i32 s9, 8                                         // 000000001794: BF048809
	s_cbranch_scc0 65456                                       // 000000001798: BFA1FFB0 <r_10_10n2+0x5c>
	s_branch 65457                                             // 00000000179C: BFA0FFB1 <r_10_10n2+0x64>
	s_load_b32 s14, s[2:3], 0x1c                               // 0000000017A0: F4000381 F800001C
	s_add_i32 s15, s6, 7                                       // 0000000017A8: 810F8706
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017AC: BF870009
	s_cmp_lt_i32 s15, 8                                        // 0000000017B0: BF04880F
	s_cbranch_scc1 65486                                       // 0000000017B4: BFA2FFCE <r_10_10n2+0xf0>
	s_load_b32 s13, s[2:3], 0x20                               // 0000000017B8: F4000341 F8000020
	s_cmp_gt_u32 s6, 0x7ffffff7                                // 0000000017C0: BF08FF06 7FFFFFF7
	s_mov_b32 s6, 0                                            // 0000000017C8: BE860080
	s_cbranch_scc0 65484                                       // 0000000017CC: BFA1FFCC <r_10_10n2+0x100>
	s_branch 65485                                             // 0000000017D0: BFA0FFCD <r_10_10n2+0x108>
