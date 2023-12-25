
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n32>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v2, 0                                        // 000000001610: 7E040280
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000001614: 84868204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b32 s10, s[2:3], null                               // 000000001624: F4000281 F8000000
	s_mov_b32 s3, 0                                            // 00000000162C: BE830080
	s_waitcnt lgkmcnt(0)                                       // 000000001630: BF89FC07
	v_subrev_f32_e64 v0, 0x5f000000, s10                       // 000000001634: D5050000 000014FF 5F000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001640: BF870121
	v_readfirstlane_b32 s11, v0                                // 000000001644: 7E160500
	v_bfrev_b32_e32 v0, 1                                      // 000000001648: 7E007081
	s_bfe_u32 s6, s11, 0x80017                                 // 00000000164C: 9306FF0B 00080017
	s_and_b32 s2, s11, 0x7fffff                                // 000000001654: 8B02FF0B 007FFFFF
	s_sub_i32 s7, 0x96, s6                                     // 00000000165C: 818706FF 00000096
	s_bitset1_b32 s2, 23                                       // 000000001664: BE821297
	s_add_i32 s8, s6, 0xffffff6a                               // 000000001668: 8108FF06 FFFFFF6A
	s_add_i32 s12, s6, 0xffffff81                              // 000000001670: 810CFF06 FFFFFF81
	s_lshr_b64 s[6:7], s[2:3], s7                              // 000000001678: 85860702
	s_lshl_b64 s[8:9], s[2:3], s8                              // 00000000167C: 84880802
	s_cmp_gt_i32 s12, 23                                       // 000000001680: BF02970C
	s_cselect_b32 s7, s9, s7                                   // 000000001684: 98070709
	s_cselect_b32 s6, s8, s6                                   // 000000001688: 98060608
	s_ashr_i32 s8, s11, 31                                     // 00000000168C: 86089F0B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001690: BF870499
	s_ashr_i32 s9, s8, 31                                      // 000000001694: 86099F08
	s_xor_b64 s[6:7], s[6:7], s[8:9]                           // 000000001698: 8D860806
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000169C: BF8704C9
	s_sub_u32 s2, s6, s8                                       // 0000000016A0: 80820806
	s_subb_u32 s6, s7, s9                                      // 0000000016A4: 82860907
	s_cmp_lt_i32 s12, 0                                        // 0000000016A8: BF04800C
	s_cselect_b32 s7, -1, 0                                    // 0000000016AC: 980780C1
	v_cndmask_b32_e64 v0, -s6, v0, s7                          // 0000000016B0: D5010000 201E0006
	s_and_b32 s6, s7, exec_lo                                  // 0000000016B8: 8B067E07
	s_cselect_b32 s8, 0, s2                                    // 0000000016BC: 98080280
	s_bfe_u32 s6, s10, 0x80017                                 // 0000000016C0: 9306FF0A 00080017
	s_and_b32 s2, s10, 0x7fffff                                // 0000000016C8: 8B02FF0A 007FFFFF
	s_sub_i32 s7, 0x96, s6                                     // 0000000016D0: 818706FF 00000096
	s_bitset1_b32 s2, 23                                       // 0000000016D8: BE821297
	s_add_i32 s9, s6, 0xffffff6a                               // 0000000016DC: 8109FF06 FFFFFF6A
	s_add_i32 s11, s6, 0xffffff81                              // 0000000016E4: 810BFF06 FFFFFF81
	s_lshr_b64 s[6:7], s[2:3], s7                              // 0000000016EC: 85860702
	s_lshl_b64 s[2:3], s[2:3], s9                              // 0000000016F0: 84820902
	s_cmp_gt_i32 s11, 23                                       // 0000000016F4: BF02970B
	s_cselect_b32 s3, s3, s7                                   // 0000000016F8: 98030703
	s_cselect_b32 s2, s2, s6                                   // 0000000016FC: 98020602
	s_ashr_i32 s6, s10, 31                                     // 000000001700: 86069F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001704: BF870499
	s_ashr_i32 s7, s6, 31                                      // 000000001708: 86079F06
	s_xor_b64 s[2:3], s[2:3], s[6:7]                           // 00000000170C: 8D820602
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001710: BF870009
	s_sub_u32 s2, s2, s6                                       // 000000001714: 80820602
	s_subb_u32 s3, s3, s7                                      // 000000001718: 82830703
	v_cmp_gt_f32_e64 s6, 0x5f000000, s10                       // 00000000171C: D4140006 000014FF 5F000000
	s_cmp_lt_i32 s11, 0                                        // 000000001728: BF04800B
	s_cselect_b32 s3, 0, s3                                    // 00000000172C: 98030380
	s_cselect_b32 s2, 0, s2                                    // 000000001730: 98020280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001734: BF870001
	v_cndmask_b32_e64 v1, v0, s3, s6                           // 000000001738: D5010001 00180700
	s_and_b32 s3, s6, exec_lo                                  // 000000001740: 8B037E06
	s_cselect_b32 s6, s2, s8                                   // 000000001744: 98060802
	s_lshl_b64 s[2:3], s[4:5], 3                               // 000000001748: 84828304
	v_mov_b32_e32 v0, s6                                       // 00000000174C: 7E000206
	s_add_u32 s0, s0, s2                                       // 000000001750: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001754: 82010301
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001758: DC6E0000 00000002
	s_nop 0                                                    // 000000001760: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001764: BFB60003
	s_endpgm                                                   // 000000001768: BFB00000
